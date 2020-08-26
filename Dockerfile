
    FROM python:3.7.3

    WORKDIR /usr/src/app

    COPY ./* ./MDSINE2
    COPY requirements.txt ./requirements.txt

    RUN pip install ./MDSINE2/PyLab/.
    RUN pip install --no-cache-dir -r requirements.txt
    WORKDIR MDSINE2
    RUN python make_real_subjset.py
    RUN mkdir output

    CMD ["python", "main_real.py", "-d", "0", "-i", "0", "-ns", "400", "-nb", "200", "-b", "output/" ]
    