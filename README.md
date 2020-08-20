# ColLa: Conversational framework for Linguistic Agents

Base:
[![](https://images.microbadger.com/badges/version/erolmatei/colla-framework-base.svg)](https://microbadger.com/images/erolmatei/colla-framework-base "Get your own version badge on microbadger.com")

Flask: [![](https://images.microbadger.com/badges/version/erolmatei/colla-framework-flask.svg)](https://microbadger.com/images/erolmatei/colla-framework-flask "Get your own version badge on microbadger.com")

## Setup

If you are cloning this repo, be sure to [install git-lfs](https://github.com/git-lfs/git-lfs/wiki/Installation) first, or you *will* have problems with dangling LFS pointers later.

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
Remember to update your `COLLA_DATADIR`

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: colla-framework-jupyter
  labels:
    app: colla-framework-jupyter
  namespace: jeffstudentsproject
spec:
  containers:
    - name: spark-jena-jupyter
      image: erolmatei/colla-framework-base:latest
      ports:
        - containerPort: 8888
          name: "jupyter-lab"
        - containerPort: 4040
          name: "spark-ui"
      resources:
        cpu: "12000m"
        memory: "16Gi"
      command:
        - 'jupyter-lab'
      args:
        - '--no-browser'
        - '--ip=0.0.0.0'
        - '--allow-root'
        - '--NotebookApp.token='
        - '--notebook-dir=/root' # Be sure to change this to /nfs when developing
      env:
        - name: COLLA_DATADIR
          value: "/nfs/colla-framework/notebooks/data" # example of custom DATADIR on a persistent volume claim
      volumeMounts:
        - mountPath: /nfs/
          name: nfs-access
  securityContext: {}
  serviceAccount: containerroot
  volumes:
    - name: nfs-access
      persistentVolumeClaim:
        claimName: jeffstudentsvol1claim # replace with yours
          
```

If you are developing it may be profitable to copy the whole source tree from `/root/colla-framework` to the volume claim and then `git checkout master`, and use `--notebook-dir=/nfs` otherwise you can't navigate to /nfs from the jupyter browser. This is useful because the volume claim is much more stable.

## Webserver setup

One needs to:

- Build the Web Application UI. This requires to install node and npm with all the required dependencies for the webapp, run the build and generate a dist folder.
Go to `webui/frontend` and run `npm install && npm run build`.

- Run the flask webserver.  Go to `webui/backend` and run `python app.py` (or use the flask command).

## Future Work

See [here](futurework.md) for details.
