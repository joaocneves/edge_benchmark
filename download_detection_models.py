import os
import urllib.request
import tarfile, sys


def untar(fname, epath):
    if (fname.endswith("tar.gz")):
        tar = tarfile.open(fname)
        tar.extractall(path=epath)
        tar.close()
        print
        "Extracted in Current Directory"
    else:
        print
        "Not a tar.gz file: '%s '" % sys.argv[0]

DOWNLOADS_DIR = 'detection_models\\'
os.mkdir(DOWNLOADS_DIR)

# For every line in the file
for url in open('detection_models.txt'):

    # Remove new-line
    url = url.rstrip()

    # Split on the rightmost / and take everything on the right side of that
    name = url.rsplit('/', 1)[-1]

    # Combine the name and the downloads directory to get the local filename
    filename = os.path.join(DOWNLOADS_DIR, name)

    # Download the file if it does not exist
    if not os.path.isfile(filename):
        urllib.request.urlretrieve(url, filename)
        untar(filename, 'detection_models')
