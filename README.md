# ColLa: Conversational framework for Linguistic Agents

Base:
[![](https://images.microbadger.com/badges/version/erolmatei/colla-framework-base.svg)](https://microbadger.com/images/erolmatei/colla-framework-base "Get your own version badge on microbadger.com")

Flask: [![](https://images.microbadger.com/badges/version/erolmatei/colla-framework-flask.svg)](https://microbadger.com/images/erolmatei/colla-framework-flask "Get your own version badge on microbadger.com")

## Description

ColLa is a simple research project about linguistic-oriented conversational agents. This project comes with a Knowledge Graph generator and a small PoC chatbot for testing purposes.

This project was started as a Summer Research Internship project by Enrico Trombetta ([@erolm-a](https://github.com/erolm-a)) and handed over to the Grill Lab.

A better explanation and overview of its architecture can be found under the [presentation](https://github.com/grill-lab/knowledge-glue/tree/master/presentation) folder.

## Setup

Setting up this project from scratch will take you some time and patience, so please grab some tea and read carefully.

If you are cloning this repo, be sure to [install git-lfs](https://github.com/git-lfs/git-lfs/wiki/Installation) first, or you *will* have problems with dangling LFS pointers later.

We assume that you are going to use this project either for regression tests or for development inside a docker container. Thus, you only need to use the right image (erolmatei/colla-framework-base) which will then invoke `setup_docker.sh`. This will, in turn, install all the required libraries.

When preparing the yaml file for launching one of the given services, you *should* specify a data folder to use. By default `/root/colla-framework/notebooks/data` will be used unless you specify the environment variable `COLLA_DATADIR` (which you will set in the YAML file for the pod, see below). You should set it to point to a volume claim folder, e.g. `/nfs/colla-framework/notebooks/data`.

You also need to download the BabelNet indices, which unfortunately cannot be automatized due to the dataset size and the machine-unfriendly download form. The indices path *should* be stored in another environment variable `COLLA_BABELNET_INDEX`. If not provided, the default value is `COLLA_BABELNET_INDEX=$COLLA_DATADIR/babelnet/BabelNet-4.0.1`.

- Register to [BabelNet](https://babelnet.org/login)
- Apply for the > 29 GB  indices dataset.
- Download it and extract it under your `$COLLA_BABELNET_INDEX`. This location should now have about 44 folders (each being a Lucene index) and 4 files.


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

Also, it may be useful to get the Spark UI to work during development. See [here](https://jupyter-docker-stacks.readthedocs.io/en/latest/using/specifics.html#apache-spark) for details.

## RDF Extraction setup

You need to generate some RDF triples in order to import them in our Fuseki server.

Inside the notebook pod we opened earlier, execute the following:

```bash
"tools/rdf_extractor.py --wiktionary-revision=20200720 --wordlist=wdpg --output=output/wdpg.ttl"
```

This commands will fetch the Wiktionary dump revision "20200720" and perform extraction and entity linking on the Wiktionary frequency list extracted from Project Gutenberg. The output will be generated at ``$COLLA_DATADIR/output/wdpg.ttl`` in Turtle format, which can be easily imported in Fuseki.

You can also perform this command as a standalone pod.

## Fuseki for KG endpoint

Unfortunately, we still do not have an automatized way to start Fuseki with a RDF triple straight away (yet), but the procedure is simple enough:

1. Use [stain/jena-fuseki](https://hub.docker.com/r/stain/jena-fuseki) as documented. You should set a password for the admin account, but it only necessary for when accessing the dashboard and making update queries.
2. Take note of the route of Fuseki. We will need it later for Flask.
3. Go to the tab "manage datasets", then press on "add new dataset", provide a dataset name (e.g. "sample_10000_common"), choose a dataset type (we recommend "persistent"), press on "create dataset". Then upload a RDF graph that you generated [earlier](#rdf-extraction-setup).
4. Take note of the name you chose for this dataset name as well.

## Webserver for PoC

The docker image [erolmatei/colla-framework-flask](https://hub.docker.com/repository/docker/erolmatei/colla-framework-flask) takes care of building up the JS blob. What you need to do is simply to create a Pod for a SPARQL provider (we'll use Fuseki as it is pretty out-of-the-box), upload a graph and then start our Flask image.

## Future Work

See [here](futurework.md) for details.
