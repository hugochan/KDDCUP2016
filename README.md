# Frontier
Frontier of Science project

Initialy this repository contains only the scripts for running a demo of the search system. It gets as input a query and outputs the top ranked documents in the database, as well as a gexf file of the subgraph induced by the given query.

## Configuration

First we need to restore the database. Download the [mysql dump](http://luamtotti.s3.amazonaws.com/public/csx.sql.bz2). Create a database on your instance of mysql (let's assume its called `csx`). Now run the following commands on the same folder of the downloaded file.

```bash
bunzip2 csx.sql.bz2
mysql -uuser -p csx < csx.sql
```

Replace `user` accordingly and enter the password when prompted.

The system uses a data folder for several purposes. **Create a data folder in any suitable location** and we'll refer to it as the **data folder** from now on (for example: /home/frontier/data).

Some pre-processed data is required to run the system (I'm currently trying to improve this). Download [this zip file](http://luamtotti.s3.amazonaws.com/public/contexts.zip) with citation contexts texts into the data folder and unpack it (`unzip contexts.zip`). Also, download [this vocabulary file](http://luamtotti.s3.amazonaws.com/public/contexts_tfidfs_tokens.txt) for the contexts and move it to the data folder. You may place or name these files differently, however you'll have to configure the paths properly in the next step.

**Open the config.py file and configure some required variables**, likely the following: DATA, DB_NAME, DB_USER, DB_PASSWD. You may configure the parameters for the method by setting entries to the PARAMS dict variable.

**Add the base folder to the python path enviroment variable**: `export PYTHONPATH=$PYTHONPATH:/home/frontier/code`. You may want to add this command to a `.bashrc` or `.pam_enviroment` file for permament effect.

On a **MAC OSX** you may have to change the socket path in `mymysql/mymysql.py` in the constructor for MyMySQL. Check the correct path for your installation  in `/etc/mysql/my.cnf`.

## Library Dependencies 

First, install the native libraries using apt-get.

```
apt-get install ant libmysqlclient-dev python-dev openjdk-7-jdk g++ make libatlas-base-dev liblapack-dev gfortran
```

Now install the python libraries using `pip`. You may have to install `pip` (python-pip on apt) and setup virtual enviroments before running this command.

```
pip install mysql-python nltk chardet numpy scipy scikit-learn networkx
```

To install PyLucene download the installation [here](http://apache.mesi.com.ar/lucene/pylucene/pylucene-4.9.0-0-src.tar.gz) and extract its contents. Make sure the variable JCC_JDK is set correctly to your JDK. In the extracted folder, run these commands:

```
cd jcc
python setup.py build
sudo python setup.py install
cd ..
# Edit Makefile to match your environment #
make
make test (look for failures)
sudo make install
```

The last step may not require sudo privileges if you are not running a global interpreter of python. You may have to set up which Java installation you want to use prior to these commands. In the **MAC OSX** it is probably easier to run `brew install pylucene` and then make sure you are using the python version used by homebrew (or add the path built libs to your working python path).

**Download the stop words** resource for NLTK: `python -m nltk.downloader stopwords`.

**Create the Lucene index**, which is going to be used for fetching files and assessing textual similarity: `python indexer/indexer.py`. The index folder path will be create according to the config file.


## Running the searcher

The code includes a command line tool to search the database for a given query and return the induced subgraph for the query. Simply run:

```bash
python ranking/run.py -q <query> [-o <gexf_file>] [-n <n_results>] [-h]
```

The graph files will be named according to the query and placed in the specified output folder.


## Pre-processing

Although users may run the search scripts directly, here we include instructions to run some of the preprocessing steps envolving
crawling, converting and parsing the publications.

The scripts can be found in the preprocess package. Most of them rely on Zeno, a simple python library for assisting in multiprocessing
and task managing. It allows process to manage which publications still need to be processed and exactly what stage on the pipeline that
publication currently is. So you can end up firing multiple process of the downloader across different machines and at the same time the
converter. They will manage to process tasks only when they are available with no overlap or race conditions. Communication across
machines is done implicitly by MySQL.

Zeno only requires MySQL configured and a database named `zeno` to start working. You can get the dump 
[here](http://luamtotti.s3.amazonaws.com/public/zeno.sql.bz2). To start each of the preprocessors, simply
provide a number of process (default 1) when running them: `python preprocess/downloader/downloader_urls.py 5`.


