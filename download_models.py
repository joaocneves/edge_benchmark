import os
import urllib.request
import tarfile, sys
import argparse

def remove_file_ext(fname):

    if fname.endswith("tar.gz"):
        return fname.replace("tar.gz", "")
    elif fname.endswith("tgz"):
        return fname.replace("tgz", "")
    else:
        return fname

def untar(fname, epath):
    if (fname.endswith("tar.gz") or fname.endswith("tgz")):
        tar = tarfile.open(fname)
        tar.extractall(path=epath)
        tar.close()
        print("Extracted in Current Directory")
    else:
        print("Not a tar.gz file: '%s '" % sys.argv[0])


def download_models(models_list, output_path, create_folder=False):

    # For every line in the file
    for url in open(models_list):

        # Remove new-line
        url = url.rstrip()

        # Split on the rightmost / and take everything on the right side of that
        name = url.rsplit('/', 1)[-1]

        # Combine the name and the downloads directory to get the local filename
        filename = os.path.join(DOWNLOADS_DIR, name)

        # Download the file if it does not exist
        if not os.path.isfile(filename):
            urllib.request.urlretrieve(url, filename)
        if create_folder:
            os.mkdir(os.path.join(output_path, remove_file_ext(name)))
            untar(filename, os.path.join(output_path, remove_file_ext(name)))
        else:
            untar(filename, output_path + '\\')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='', help='the task to which the model was built to')
    args = parser.parse_args()

    if args.model_type == 'Detection':
        DOWNLOADS_DIR = 'detection_models\\'
        if not os.path.exists(DOWNLOADS_DIR):
            os.mkdir(DOWNLOADS_DIR)
        download_models('info\\detection_models.txt', DOWNLOADS_DIR, create_folder=False)
    elif args.model_type == 'Classification':
        DOWNLOADS_DIR = 'general_models\\'
        if not os.path.exists(DOWNLOADS_DIR):
            os.mkdir(DOWNLOADS_DIR)
        download_models('info\\general_models.txt', DOWNLOADS_DIR, create_folder=True)
    else:
        print('Wrong args')
