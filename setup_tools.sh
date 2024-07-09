#!/bin/bash

# Update package list
sudo apt-get update

# Install dependencies
sudo apt-get install -y gnupg software-properties-common curl

# Add HashiCorp GPG key
curl -fsSL https://apt.releases.hashicorp.com/gpg | sudo apt-key add -

# Add Terraform repository
sudo apt-add-repository "deb [arch=amd64] https://apt.releases.hashicorp.com $(lsb_release -cs) main"

# Update package list again
sudo apt-get update

# Install Terraform
sudo apt-get install -y terraform

# Install AWS CLI
sudo apt-get update
sudo apt-get install -y awscli

# Install Ansible
sudo apt-get update
sudo apt-get install -y software-properties-common
sudo add-apt-repository --yes --update ppa:ansible/ansible
sudo apt-get install -y ansible

# Verify installation
terraform --version

# Verify installation
aws --version

# Verify Ansible installation
ansible --version
