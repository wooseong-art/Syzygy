#!/usr/bin/env python3
"""Conformance submitter: verify the kit, zip it, fetch the upload form, upload.

A pure-Python stand-in for submit_conformance.sh, for Windows where the
curl/MSYS upload path has failed. Needs only Python 3 -- no bash, zip or jq.

  python submit.py            # verify, zip, upload
  python submit.py check      # verify and zip only, no upload
  python submit.py status     # show submission status and build log links

Reads PORTAL from .team next to this script.
"""

import hashlib
import json
import os
import re
import sys
import urllib.error
import urllib.request
import uuid
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
KIT = os.path.join(HERE, "conformance_pack")
ZIP = os.path.join(HERE, "kit.zip")
SKIP = {".DS_Store", "Thumbs.db"}


def die(msg):
    print("ERROR: %s" % msg, file=sys.stderr)
    sys.exit(1)


def portal_url():
    """Pull PORTAL out of .team without needing a shell to source it."""
    path = os.path.join(HERE, ".team")
    if not os.path.exists(path):
        die(".team not found. Put your portal link in it as PORTAL=\"https://...\"")
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            if k.strip() == "PORTAL":
                v = v.strip().strip('"').strip("'")
                if not v.startswith("http"):
                    die("PORTAL in .team is empty or is not a link.")
                return v
    die("no PORTAL line in .team")


def kit_files():
    """Every kit file, as (relative posix path, absolute path), sorted."""
    out = []
    for root, _, names in os.walk(KIT):
        for name in names:
            if name in SKIP:
                continue
            full = os.path.join(root, name)
            rel = os.path.relpath(full, KIT).replace(os.sep, "/")
            if rel.startswith("__MACOSX/"):
                continue
            out.append((rel, full))
    return sorted(out)


def verify():
    """Reproduce verify.sh's manifest digest, reading its expected value."""
    with open(os.path.join(KIT, "verify.sh"), encoding="utf-8") as fh:
        m = re.search(r'EXPECTED="([0-9a-f]{64})"', fh.read())
    if not m:
        die("could not read EXPECTED from conformance_pack/verify.sh")
    expected = m.group(1)

    buf = b""
    for rel, full in kit_files():
        if rel == "verify.sh":          # verify.sh excludes itself
            continue
        with open(full, "rb") as fh:
            digest = hashlib.sha256(fh.read()).hexdigest()
        buf += rel.encode() + b"\x00" + digest.encode() + b"\n"
    got = hashlib.sha256(buf).hexdigest()

    print("expected: %s" % expected)
    print("got:      %s" % got)
    if got != expected:
        die("MODIFIED - re-download the kit. Do not try to repair it by hand.")
    print("OK - kit is unmodified")


def make_zip():
    """Zip the kit's CONTENTS (not the folder) at the archive root."""
    if os.path.exists(ZIP):
        os.remove(ZIP)
    with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
        for rel, full in kit_files():
            zf.write(full, rel)
    names = zipfile.ZipFile(ZIP).namelist()
    print("kit.zip: %d bytes, %d files at top level" % (os.path.getsize(ZIP), len(names)))
    print("  " + "  ".join(names))


def fetch_portal(url):
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        die("portal returned HTTP %d. A 403 means the PORTAL link is wrong or "
            "cut off." % e.code)
    except Exception as e:                                   # noqa: BLE001
        die("could not reach the portal: %s" % e)


def upload(form):
    url, fields = form["url"], form["fields"]
    print("url:    %s" % url)
    print("fields: %s" % ", ".join("%s(%d)" % (k, len(v)) for k, v in fields.items()))

    bnd = "----py" + uuid.uuid4().hex
    body = bytearray()
    for k, v in fields.items():                  # keep the portal's own order
        body += b"--" + bnd.encode() + b"\r\n"
        body += ('Content-Disposition: form-data; name="%s"\r\n\r\n' % k).encode()
        body += v.encode() + b"\r\n"

    with open(ZIP, "rb") as fh:                  # 'file' MUST come last: S3
        data = fh.read()                         # ignores anything after it
    body += b"--" + bnd.encode() + b"\r\n"
    body += b'Content-Disposition: form-data; name="file"; filename="kit.zip"\r\n'
    body += b"Content-Type: application/zip\r\n\r\n" + data + b"\r\n"
    body += b"--" + bnd.encode() + b"--\r\n"

    req = urllib.request.Request(
        url, data=bytes(body), method="POST",
        headers={"Content-Type": "multipart/form-data; boundary=" + bnd})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            print("\nHTTP %d - your zip arrived." % r.status)
    except urllib.error.HTTPError as e:
        print("\nHTTP %d - the upload was rejected:" % e.code, file=sys.stderr)
        print(e.read().decode("utf-8", "replace")[:900], file=sys.stderr)
        die("your upload form lasts 1 hour, so just run this again.")
    print("The build takes a few minutes. Then run:  python submit.py status")


def show_status(doc):
    keys = ("submission_id", "eligibility", "conformance", "conformance_at",
            "failure_reason")
    print(json.dumps({k: doc.get(k) for k in keys}, indent=2))
    logs = [x.get("url") for x in (doc.get("logs") or []) if x.get("url")]
    if logs:
        print("\nBuild log links (valid for 1 hour):")
        for u in logs:
            print(u)


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "submit"
    if cmd not in ("submit", "check", "status"):
        die("usage: python submit.py [check|status]")

    if cmd == "status":
        show_status(fetch_portal(portal_url()))
        return

    print("1/4 Checking the kit is unmodified...")
    verify()
    print("\n2/4 Zipping...")
    make_zip()
    if cmd == "check":
        print("\ncheck only - nothing uploaded.")
        return
    print("\n3/4 Getting your upload form from the portal...")
    doc = fetch_portal(portal_url())
    form = doc.get("upload")
    if not form or not form.get("url") or not form.get("fields"):
        die("the portal did not return an upload form.")
    print("\n4/4 Uploading kit.zip...")
    upload(form)


if __name__ == "__main__":
    main()
