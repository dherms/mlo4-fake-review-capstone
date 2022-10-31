data "aws_ami" "ubuntu" {
  most_recent = true

  filter {
    name   = "name"
    values = ["ubuntu/images/hvm-ssd/*20.04-amd64-server-*"]
  }
  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }
  owners = ["099720109477"] # Canonical
}

resource "aws_default_vpc" "default_vpc" {
}

resource "aws_key_pair" "infrastructure-key" {
  key_name   = "fake-review-infra-key"
  public_key = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAICro0h9xjs/4cwgKh5uym4bdl45lYfft00zDPpPpP56Y fake-review-infra-key"
}

resource "aws_security_group" "model_tracking_server_sg" {
  name        = "model_tracking_server_sg"
  description = "Allow inbound traffic to MLFlow Model Tracking Server"
  vpc_id      = aws_default_vpc.default_vpc.id

  ingress {
    description = "TLS to MLFlow port"
    from_port   = 5000
    to_port     = 5000
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    description = "SSH"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port        = 0
    to_port          = 0
    protocol         = "-1"
    cidr_blocks      = ["0.0.0.0/0"]
    ipv6_cidr_blocks = ["::/0"]
  }
}

resource "aws_instance" "mlflow_model_tracking_server" {
  ami                    = data.aws_ami.ubuntu.id
  instance_type          = "t3.medium"
  key_name               = aws_key_pair.infrastructure-key.key_name
  vpc_security_group_ids = [aws_security_group.model_tracking_server_sg.id]

  user_data = <<EOF
#!/bin/bash
sudo apt-get update -y && sudo apt-get upgrade -y
sudo apt-get install python3-pip -y
pip3 install --upgrade --force-reinstall click black
pip3 install mlflow
mlflow server \
  --default-artifact-root s3://fourthbrain-fake-review-detector/mlflow/artifacts/ \
  --backend-store-uri sqlite:///mlflowdb.sqlite --host 0.0.0.0 --port 5000
EOF


  tags = {
    Name = "MLFlow Model Registry"
  }
}
