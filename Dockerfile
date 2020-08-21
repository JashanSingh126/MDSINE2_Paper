FROM python:3.7.3

WORKDIR /usr/src/app

COPY ./PyLab ./PyLab
COPY MDSINE2/requirements.txt ./requirements.txt
COPY ./MDSINE2 ./MDSINE2

RUN pip install --no-cache-dir -r requirements.txt
RUN pip install ./PyLab/.
WORKDIR MDSINE2
RUN python make_real_subjset.py

CMD ["python", "main_real.py", "-d", "5", "-i", "0", "-ns", "200", "-nb", "100", "-b", "output/" ]