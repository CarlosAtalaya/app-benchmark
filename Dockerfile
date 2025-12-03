FROM continuumio/miniconda3:25.3.1-1

WORKDIR /app

# Instalamos dependencias del sistema
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY environmentlite.yml .

# Capa pesada (YA CACHEADA, no se repetirá)
RUN conda env create -f environmentlite.yml \
    && conda run -n mapfre pip cache purge \
    && conda clean -afy \
    && rm -rf /root/.cache/pip

# Fix de librerías (YA CACHEADO)
RUN echo "/opt/conda/envs/mapfre/lib" > /etc/ld.so.conf.d/conda.conf && ldconfig
RUN cd /opt/conda/envs/mapfre/lib && \
    ln -sf libmkl_intel_lp64.so libmkl_intel_lp64.so.1 && \
    ln -sf libmkl_gnu_thread.so libmkl_gnu_thread.so.1 && \
    ln -sf libmkl_core.so libmkl_core.so.1

# --- NUEVO: INSTALACIÓN RÁPIDA DE PYCOCOTOOLS ---
# Lo hacemos aquí para no invalidar la caché de las capas anteriores
# -----------------------------------------------
RUN conda run -n mapfre pip install pycocotools ftfy regex

SHELL ["conda", "run", "-n", "mapfre", "/bin/bash", "-c"]

COPY . .

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "mapfre"]
CMD ["python", "main.py"]