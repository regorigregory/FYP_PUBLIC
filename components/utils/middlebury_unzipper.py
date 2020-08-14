# inspiration was drawn from the comments at https://stackoverflow.com/questions/3451111/unzipping-files-in-python.
import zipfile
import glob
import os


class MiddleburyUnzipper():
    def __init__(self, rootPath):
        print("Constructor has been called")
        self.rootPath = rootPath
        self.extractPath = rootPath
        self.zipFiles = glob.glob(os.path.join(self.rootPath, "*.zip"), recursive = True)

    def unzipFiles(self):
        for zippedPath in self.zipFiles:
             with zipfile.ZipFile(zippedPath, 'r') as zip_ref:
                zip_ref.extractall(self.extractPath)
                print("%s file has been unzipped!"%zippedPath)