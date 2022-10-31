output "mlflow_server_ip" {
  description = "Public IP of MLFlow Server"
  value       = aws_instance.mlflow_model_tracking_server.public_ip
}