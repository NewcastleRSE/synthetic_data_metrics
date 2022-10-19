## Repo setup
* Come up with your package name 
* Create folder for project files, echoing repo name.
* Create `__init__.py`. we need this if we want to import files in this package elsewhere.
* Create project file. When someone uses this package, they'll use the name of this file when importing.

## Virtual environment
There's plenty of discussion about the best way to manage a virtual environment for package developemt, but there's a generally positive opinion of using `venv`. This comes installed with Python since Python 3.3. `venv` installs `pip` into the virtual environment by default. 

Create your environment (calling it whatever you want, in this case, `synth-data-env`):

`python3 -m venv synth-data-env`

Activate your environment:
`source synth-data-env/bin/activate`

You can leave it active to complete the following commands, and then when you are finished working run:
`deactivate`


## Requirements
Create `requirements_dev.txt` and `requirements.txt`. We'll use this to describe packages that contributors and users need to install. Add `pip` and `wheel` to the dev requirements. As you develop you might want to use the latest versions of packages, and update them as you go. Once you release your package you can stipulate the exact version contributors should use (using `==`), the minimum version (using `>=`) or a compatible release (`~=`).

Install requirements:
```
pip install -r requirements_dev.txt
```

or 

```
pip3 install -r requirements_dev.txt
```

## Create a test function
In the project file you created above, create a simple test function. Use 3 double or single quotation marks to create a docstring that describes the purpose, arguments and return value of the function.

```
def function(a):
    '''
    Describe function's purpose.
    Args:
        a (str): word to do something with
    Returns:
        None
    '''
    
    function contents
```

## Licence
Create a licence. I've used MIT here, and included the required text in `LICENSE`.


## Create a build script
Next we'll create a script for building our package to upload to PyPI (Python package index). See `setup.py` for an example and this [link](https://packaging.python.org/en/latest/guides/distributing-packages-using-setuptools/) for documentation of additional parameters you can use. This [site](http://turbo87.github.io/setup.py/) provides a more interactive guide.

Build your package using this command: `python3 setup.py sdist bdist_wheel`. This builds a package ready for distribution. You can read an explanation of these commands [here](https://medium.com/ochrona/understanding-python-package-distribution-types-25d53308a9a). You should now see `dist`, `egg-info`and `build` folders.


## Publish to TestPyPI
`TestPyPI` lets you test your package before releasing publically. You'll need to [register](https://test.pypi.org/account/register/) for an account (this is a seperate registration to your PyPI account if you have one). We will use `Twine` to publish our package using this command:

`twine upload --repository-url https://test.pypi.org/legacy/ dist/*`

You will then be prompted for your log in details. If there's any errors here, fix them, delete the folders you created in the build step and rerun the build and twine commands. 

Once this runs successfully yu will be able to view the package online.

## Using test package
In a new terminal, create a virtual environment as above and install your test package using the command provided on the package web page which will look like this:

`pip install -i https://test.pypi.org/simple/ my_package`

If your package requires packages available from the 'real' PyPI, use this format instead:

`pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple my_package`

Import your function and use it:

```
from synthmetricstet.to_lower_case import process
process('CAPS')
```

## Publishing for real
To push your code for real use this command:
`twine upload dist/*`.

Don't forget to update the version number in the setup file each time you do a new version.

