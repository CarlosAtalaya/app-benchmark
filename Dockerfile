FROM continuumio/miniconda3:25.3.1-1

WORKDIR /app

# Instalamos dependencias del sistema
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY environmentlite.yml .
RUN conda env create -f environmentlite.yml && conda clean -afy

# --- AGREGAR ESTA LÍNEA AQUÍ ---
# Aseguramos que el sistema encuentre las librerías C++ del entorno conda mapfre (incluyendo MKL)
ENV LD_LIBRARY_PATH /opt/conda/envs/mapfre/lib:$LD_LIBRARY_PATH

SHELL ["conda", "run", "-n", "mapfre", "/bin/bash", "-c"]

COPY . .
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "mapfre"]

CMD ["python", "main.py"] 
# Nota: He cambiado script_principal.py por main.py ya que es el archivo que subiste.