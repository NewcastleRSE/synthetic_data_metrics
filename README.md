# Standard Project
A template repo for the standard RSE project

## About

Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed sollicitudin ante at eleifend eleifend. Sed non vestibulum nisi. Aliquam vel condimentum quam. Donec fringilla et purus at auctor. Praesent euismod vitae metus non consectetur. Sed interdum aliquet nisl at efficitur. Nulla urna quam, gravida eget elementum eget, mattis nec tortor. Fusce ut neque tellus. Integer at magna feugiat lacus porta posuere eget vitae metus.

Curabitur a tempus arcu. Maecenas blandit risus quam, quis convallis justo pretium in. Suspendisse rutrum, elit at venenatis cursus, dolor ligula iaculis dui, ut dignissim enim justo at ligula. Donec interdum dignissim egestas. Nullam nec ultrices enim. Nam quis arcu tincidunt, auctor purus sit amet, aliquam libero. Fusce rhoncus lectus ac imperdiet varius. Sed gravida urna eros, ac luctus justo condimentum nec. Integer ultrices nibh in neque sagittis, at pretium erat pretium. Praesent feugiat purus id iaculis laoreet. Proin in tellus tristique, congue ante in, sodales quam. Sed imperdiet est tortor, eget vestibulum tortor pulvinar volutpat. In et pretium nisl.

## Some suggested reading about managing the project

Code should be tested and have automated tests that run on every push to the main branch at the least. Some useful links to get started:

* [GitHub Actions](https://github.com/features/actions)
* [Testing ML projects](https://neptune.ai/blog/automated-testing-machine-learning)

### Project Team
Dr L. Ipsum, Newcastle University  ([lorem.ipsum@newcastle.ac.uk](mailto:lorem.ipsum@newcastle.ac.uk))  
Professor D. Sit Amet, XY University  ([d.sit.amet@newcastle.ac.uk](mailto:d.sit.amet@example.com))  

### RSE Contact
Dr Kate Court
RSE Team  
Newcastle University  
([kate.court@newcastle.ac.uk](mailto:kate.court@newcastle.ac.uk))  

## Built With

This section is intended to list the frameworks and tools you're using to develop this software. Please link to the home page or documentatation in each case.

[Framework 1](https://something.com)  
[Framework 2](https://something.com)  
[Framework 3](https://something.com)  

## Getting Started

### Prerequisites

Any tools or versions of languages needed to run code. For example specific Python or Node versions. Minimum hardware requirements also go here.

### Installation

Install Python. Instructions to follow.

### Running Locally

After cloning the repo into a new directory, make a virtual environment, activate it, and install the dependencies, e.g.

```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements_dev.txt
```

Once finished   

```
deactivate
```

### Running Tests

#### Linting
Use `flake8` to identify formatting errors in your code **before** pushing. 

```
flake8 path/to/file/to/test
```

## Deployment

### Production

See [creating_a_python_package.md](creating_a_python_package.md).

## Usage

Any links to production environment, video demos and screenshots.

## Roadmap

- [x] Initial Research  
- [ ] Minimum viable product
- [ ] Alpha Release  
- [ ] Feature-Complete Release  

## Contributing

### Issues
In GitHub issues represent problems, ideas or features that you need to work on. You can create them by going to the Issues tab on the GitHub website. Each issue has a number, a title and then a space for comments if needed. You can add comments to further descrube what needs doing, track how you are intending to do the work or keep track of things that you have tried that didn't work. When you create an issue, use the option on the right hand side of the page to add it to the project for this repo (see below) and then select which column to use.

You can refer to an issue in a commit by including the issue number preceded by a hash in the commit message. If you open the issue relevant commits will be listed. You can add tags or labels to issues, for example to distinguish bugs or updates. Once an issue is completed you should close it.

You don't need to track every single task you do as an issue, however one of the most useful features is the ability to assign an issue to a particular user so it is worth adding most areas of work. This means you can mark issues you intend to work on or divide work between you.

### Projects
GitHub has a Kanban Board feature called 'Projects'. Find the project board for this repo under 'Projects'. You can add issues to the columns, currently Todo, In Progress and Done. You can add more problems (e.g. help wanted) if you need. This adds a layer of functionality on top of issues enabling you to see not only what has been assigned to who, but who has started worked on what.

### New metrics
All new  metrics should be accompanied by documentation describing:
* What type of data has this metric been designed for (aim for as broad as possible)?
* Where can the test data be found (prefer publically available sources and provide download instructions)?
* How should the new code be used?

Additionally, new code should be tested and, where new tests have been written, these should run automatically using GitHub Actions if possible. The process for this has yet to be confirmed.

All code should be well commented, and public functions and modules should be documented using [docstrings](https://realpython.com/documenting-python-code/). Package docstrings should be included at the top of the `_init_.py` file. 

### Main Branch
Protected and can only be pushed to via pull requests. Should be considered stable and a representation of production code. Additional rules can be created in Settings > Branches. 

### Dev Branch
Should be considered fragile, code should compile and run but features may be prone to errors.

### Feature Branches
A branch per feature being worked on.

https://nvie.com/posts/a-successful-git-branching-model/

### GitHub Actions
There is currently one workflow called `lint.yaml` that must successfully complete for a pull request to pass. This runs `flake8` linting. If your run fails, check the logs to see what error(s) linting has identified, fix your code, check linting passes locally, push your changes and create a new pull request.

## License

## Citiation

Please cite the associated papers for this work if you use this code:

```
@article{xxx2021paper,
  title={Title},
  author={Author},
  journal={arXiv},
  year={2021}
}
```


## Acknowledgements
This work was funded by The Turing Institute.
