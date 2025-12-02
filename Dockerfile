FROM continuumio/miniconda3:25.3.1-1

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY environmentlite.yml .

RUN conda env create -f environmentlite.yml && conda clean -afy

SHELL ["conda", "run", "-n", "mapfre", "/bin/bash", "-c"]

COPY . .

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "mapfre"]

CMD ["python", "script_principal.py"]