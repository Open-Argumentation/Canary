import os
import zipfile
from pathlib import Path

import requests

from canary import logger
from canary.utils import MODEL_DOWNLOAD_LOCATION, MODEL_STORAGE_LOCATION


def download_pretrained_models(model_id, location=None, download_to=None, overwrite=False):
    if model_id is None:
        raise ValueError("model_id cannot be empty")

    if location is None:
        location = MODEL_DOWNLOAD_LOCATION

    if download_to is None:
        download_to = Path(MODEL_STORAGE_LOCATION / "models.zip")

    # Let's get the pretrained stuff
    if os.path.isfile(download_to) is False or overwrite is True:
        response = requests.get(location, stream=True)
        if response.status_code == 200:
            content_type = response.headers.get('Content-Type')
            logger.debug("Models downloaded!")

            if content_type == "application/zip":
                with open(download_to, "wb") as f:
                    f.write(response.raw.read())
                    f.close()
                with zipfile.ZipFile(download_to, "r") as zf:
                    zf.extractall(Path(MODEL_STORAGE_LOCATION))
            else:
                logger.error("Didn't find a zip file.")

