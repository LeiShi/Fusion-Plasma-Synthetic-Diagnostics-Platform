# Steps to make a release

1. make sure every source file is in the right form for release

2. modify the AUTHORS, LICENSE, NOTICE files if needed

3. modify the setup.py file to include all the new metadata: new classifiers, data files, packages, modules, extensions, etc.

4. commit the changes, tag it with the desired version number.

5. run "python setup.py sdist" and "python setup.py bdist_wheel" to get the source distribution and binary distribution.

6. commit them with LFS, and merge to the master. 

7. update the Readme.md file and any documentations. 
