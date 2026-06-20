# Select base image
FROM python:3.12-slim

# Set workng directory
WORKDIR /app

# Copy only the requirements
COPY requirements.txt .

# Installment
RUN pip install --no-cache-dir -r requirements.txt

# Copy all the rest of the code
COPY . .

# Set workng directory
WORKDIR /app/src

# Statement about the container's port (no port opens)
EXPOSE 8000

# Starting command
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
