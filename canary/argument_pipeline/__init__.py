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

    if location is None:
        location = MODEL_DOWNLOAD_LOCATION

    if download_to is None:
        download_to = Path(MODEL_STORAGE_LOCATION / "models.zip")

    # Let's get the pretrained stuff
    if os.path.isfile(download_to) is False or overwrite is True:
        response = requests.get(location, stream=True)
        if response.status_code == 200:
            with open(download_to, "wb") as f:
                f.write(response.raw.read())
                f.close()

            with zipfile.ZipFile(download_to, "r") as zf:
                zf.extractall(Path(MODEL_STORAGE_LOCATION))
                logger.debug("Models downloaded!")
