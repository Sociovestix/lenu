
<h1 align="center">
lenu - Legal Entity Name Understanding 
</h1>

---------------

<h1 align="center">
<a href="https://gleif.org">
<img src="http://sdglabs.ai/wp-content/uploads/2022/07/gleif-logo-new.png" width="220" alt="">
</a>
</h1><br>
<h3 align="center">in collaboration with</h3> 
<h1 align="center">
<a href="https://sociovestix.com">
<img src="https://sociovestix.com/img/svl_logo_centered.svg" width="700px">
</a>
</h1><br>

---------------

[![License](https://img.shields.io/github/license/Sociovestix/lenu.svg)](https://github.com/Sociovestix/lenu/blob/main/LICENSE)
![](https://img.shields.io/badge/python-3.8%20%7C%203.9-blue)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)


**lenu** is a python library that helps to understand and work with Legal Entity Names
in the context of the [Legal Entity Identifier](https://www.gleif.org/en/about-lei/introducing-the-legal-entity-identifier-lei) (LEI) Standard (ISO 17441)
as well as the [Entity Legal Form (ELF) Code List](https://www.gleif.org/en/about-lei/code-lists/iso-20275-entity-legal-forms-code-list) Standard (ISO 20275).  

The library utilizes Machine Learning with Transformers and scikit-learn. It provides and utilizes pre-trained ELF Detection models published at https://huggingface.co/Sociovestix. This code as well as the LEI data and models are distributed under Creative Commons Zero 1.0 Universal license.

The project was started in November 2021 as a collaboration of the [Global Legal Entity Identifier Foundation](https://gleif.org) (GLEIF) and
[Sociovestix Labs](https://sociovestix.com) with the goal to explore how Machine Learning can support in detecting the legal form (ELF Code) from a legal name. 

It provides:
- an interface to download [LEI](https://www.gleif.org/en/lei-data/gleif-golden-copy/download-the-golden-copy#/) and [ELF Code](https://www.gleif.org/en/about-lei/code-lists/iso-20275-entity-legal-forms-code-list) data from GLEIF's public website
- an interface to train and make use of Machine Learning models to classify ELF Codes from given Legal Names
- an interface to use pre-trained ELF Detection models published on https://huggingface.co/Sociovestix
---

## Dependencies
**lenu** requires
- python (>=3.8, <3.10)
- [scikit-learn](https://scikit-learn.org/) - Provides Machine Learning functionality for token based modelling
- [transformers](https://huggingface.co/docs/transformers/index) - Download and applying Neural Network Models
- [pytorch](https://pytorch.org/) - Machine Learning Framework to train Neural Network Models
- [pandas](https://pandas.pydata.org/) - For reading and handling data
- [Typer](https://typer.tiangolo.com/) - Adds the command line interface
- [requests](https://docs.python-requests.org/en/latest/) and [pydantic](https://pydantic-docs.helpmanual.io/) - For downloading LEI data from GLEIF's website

## Installation

via PyPI:
```shell
pip install lenu
```

From github:
```shell
pip install https://github.com/Sociovestix/lenu
```

Editable install from locally cloned repository
```shell
git clone https://github.com/Sociovestix/lenu
pip install -e lenu
```

## Usage

Create folders for LEI and ELF Code data and to store your models

```shell
mkdir data
mkdir models
```

Download LEI data and ELF Code data into your `data` folder
```shell
lenu download
```

Train a (default) ELF Code Classification model. An ELF Classification model is always Jurisdiction specific and 
will be trained from Legal Names from this Jurisdiction.

Examples: 
```shell
lenu train DE       # Germany
lenu train US-DE    # United States - Delaware
lenu train IT       # Italy
```

Identify ELF Code by using a model. The tool will return the best scoring ELF Codes. 
```shell
lenu elf DE "Hans Müller KG"
#   ELF Code                  Entity Legal Form name Local name     Score
# 0     8Z6G                              Kommanditgesellschaft  0.979568
# 1     V2YH                       Stiftung des privaten Rechts  0.001141
# 2     OL20  Einzelunternehmen, eingetragener Kaufmann, ein...  0.000714
```

You can also use pre-trained models, which is recommended in most cases:
```shell
# Model available at https://huggingface.co/Sociovestix/lenu_DE
lenu elf Sociovestix/lenu_DE "Hans Müller KG"  
#  ELF Code      Entity Legal Form name Local name     Score
#0     8Z6G                  Kommanditgesellschaft  0.999445
#1     2HBR  Gesellschaft mit beschränkter Haftung  0.000247
#2     FR3V       Gesellschaft bürgerlichen Rechts  0.000071
```

## Support and Contributing
Feel free to reach out to either [Sociovestix Labs](https://sociovestix.com/contact) or [GLEIF](https://www.gleif.org/contact/contact-information)
if you need support in using this library, in utilizing LEI data in general, or in case you would like to contribute to this library in any form.
