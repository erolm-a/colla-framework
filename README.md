# knowledge-glue

## Setup

We assume that you are going to use this project either for regression tests or for development inside a docker container. Thus, you only need to use the right image (erolmatei/colla-framework-base) which will then invoke `setup_docker.sh`. This will, in turn, install all the required libraries.

When preparing the yaml file for launching one of the given services, you *should* specify a data folder to use. By default `/root/colla-framework/notebooks/data` will be used unless you specify the environment variable `COLLA_DATADIR` (which you will set in the YAML file for the pod, see below). You should set it to point to a volume claim folder, e.g. `/nfs/colla-framework/notebooks/data`.

You also need to download the BabelNet indices, which unfortunately cannot be automatized due to the dataset size and the machine-unfriendly download form. The indices path *should* be stored in another environment variable `COLLA_BABELNET_INDEX`. If not provided, the default value is `COLLA_BABELNET_INDEX=$COLLA_DATADIR/babelnet/BabelNet-4.0.1`.

- Register to [BabelNet](https://babelnet.org/login)
- Apply for the > 29 GB  indices dataset.
- Download it and extract it under your `$COLLA_BABELNET_INDEX`. This location should now have about 44 folders (each being a Lucene index) and 4 files.


## RDF Extraction setup

TODO

## Jupyter Development/Testing Pod

This pod is useful for running the notebooks.

```yaml

```

If you are developing it may be profitable to copy the whole source tree from `/root/colla-framework` to the volume claim and then `git checkout master`.


## Webserver setup

One needs to:

- Build the Web Application UI. This requires to install node and npm with all the required dependencies for the webapp, run the build and generate a dist folder.
Go to `webui/frontend` and run `npm install && npm run build`.

- Run the flask webserver.  Go to `webui/backend` and run `python app.py` (or use the flask command).

## Future Work

See [here](futurework.md) for details.