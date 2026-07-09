# AIOT Global Production Framework - GitHub Deployment Script
# Run this script to securely push your PhD framework to a remote GitHub repository.

Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "  AQIP GitHub Deployment Initializer" -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan

$repoUrl = Read-Host "Please enter your GitHub Repository URL (e.g., https://github.com/YourName/AQIP.git)"

if ([string]::IsNullOrWhiteSpace($repoUrl)) {
    Write-Host "Error: Repository URL cannot be empty. Aborting." -ForegroundColor Red
    exit
}

Write-Host "Initializing local Git repository..." -ForegroundColor Yellow
git init

Write-Host "Adding all framework components..." -ForegroundColor Yellow
git add .

Write-Host "Committing Framework..." -ForegroundColor Yellow
git commit -m "AIOT Global Production Framework: All-in-One Deployment (v7.0)"

Write-Host "Setting remote branch..." -ForegroundColor Yellow
git branch -M main
git remote add origin $repoUrl

Write-Host "Pushing to GitHub..." -ForegroundColor Yellow
git push -u origin main

Write-Host "=========================================" -ForegroundColor Green
Write-Host "  Deployment Successful!" -ForegroundColor Green
Write-Host "  Your PhD Framework is now securely backed up on GitHub." -ForegroundColor Green
Write-Host "=========================================" -ForegroundColor Green
