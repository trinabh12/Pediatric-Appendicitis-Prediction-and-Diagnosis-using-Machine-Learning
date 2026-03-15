FROM python:3.11-slim

# 2. Install system-level bouncers for OpenCV
# OpenCV needs these Linux libraries to handle images
# 2. Install system-level dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# 3. Set the working directory
WORKDIR /app

# 4. Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy the entire project into the container
COPY . .

# 6. Set the environment variable for the Port
ENV PORT=8000

# 7. Fire it up!
# We run main.py from the api folder
CMD ["python", "api/main.py"]