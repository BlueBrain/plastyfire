"""
Small helper functions to submit jobs to Slurm from Python
authors: Giuseppe Chindemi (12.2020) + minor modifications and docs by András Ecker (05.2024)
"""


import re
import time
import random
import logging
import subprocess

logger = logging.getLogger(__name__)


def submitjob(block=True):
    """Submit job and get its ID"""
    time.sleep(random.randint(0, 60))
    job_out = subprocess.run(["sbatch", "--parsable", "ipp.sh"], check=True, text=True, stdout=subprocess.PIPE).stdout
    jobid = int(re.findall(r'\d+', job_out)[0])
    logger.debug("Submitted job %i", jobid)
    if block:
        # Wait for job to start running
        while True:
            logger.debug("Waiting for job to start")
            job_info = subprocess.run(["scontrol", "show", "job", "%i" % jobid, "-o"],
                                      check=True, text=True, stdout=subprocess.PIPE).stdout
            status = re.findall(r"JobState=(\w*)", job_info)[0]
            logger.debug("Job status = %s", status)
            if status == "RUNNING":
                break
            else:  # try again in a bit
                time.sleep(10)
    return jobid


def canceljob(jobid):
    subprocess.run(["scancel", "%i" % jobid], check=True, text=True, stdout=subprocess.PIPE).stdout