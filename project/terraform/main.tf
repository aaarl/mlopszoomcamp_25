provider "aws" {
  region = var.aws_region
}

resource "aws_ecr_repository" "breast_cancer_repo" {
  name = "breast-cancer-api"
}

resource "aws_instance" "api_server" {
  ami           = "ami-0c55b159cbfafe1f0"
  instance_type = "t2.micro"
}
