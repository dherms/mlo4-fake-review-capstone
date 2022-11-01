terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~>4.16"
    }
  }

  backend "s3" {
    bucket = "fourthbrain-fake-review-detector"
    key    = "terraform"
    region = "us-west-2"
  }
}

provider "aws" {
  default_tags {
    tags = {
      Team = "Fake Review Detector"
    }
  }
  region = "us-west-2"
}
