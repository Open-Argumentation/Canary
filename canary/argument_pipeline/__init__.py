import os
import zipfile
import requests
from pathlib import Path
from canary import logger
from canary.utils import MODEL_DOWNLOAD_LOCATION, MODEL_STORAGE_LOCATION


def download_pretrained_models(location=None, download_to=None, overwrite=False):
    """
    Download the pretrained models from a web location

    :param location: the url of the model zip archive
    :param download_to: an alternative place to download the models
    :param overwrite: overwrite existing models
    """

    # Make sure the canary local directory(s) exist beforehand
    os.makedirs(MODEL_STORAGE_LOCATION, exist_ok=True)

    def unzip_models():
        with zipfile.ZipFile(download_to, "r") as zf:
            zf.extractall(Path(MODEL_STORAGE_LOCATION))
            logger.debug("models extracted.")

    if location is None:
        location = MODEL_DOWNLOAD_LOCATION

    if download_to is None:
        download_to = Path(MODEL_STORAGE_LOCATION / "models.zip")

    model_zip_present = os.path.isfile(download_to)

    # Let's get the pretrained stuff
    if model_zip_present is False or overwrite is True:
        response = requests.get(location, stream=True)

        if response.status_code == 200:
            with open(download_to, "wb") as f:
                f.write(response.raw.read())
                f.close()
                logger.debug("Models downloaded! Attempting to unzip.")

            unzip_models()
        else:
            logger.error(f"There was an issue getting the models. Http response code: {response.status_code}")
    elif model_zip_present is True:
        logger.debug("model zip already present.")
        unzip_models()
