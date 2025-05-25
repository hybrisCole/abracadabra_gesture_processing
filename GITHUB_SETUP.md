# Setting Up GitHub Remote

Follow these steps to push your local repository to GitHub:

## 1. Create a new repository on GitHub

1. Go to [GitHub](https://github.com/) and sign in
2. Click the "+" icon in the top right and select "New repository"
3. Name your repository (e.g., "atomic-movement-detection")
4. Choose public or private
5. Do not initialize with a README, .gitignore, or license (as we've already done this locally)
6. Click "Create repository"

## 2. Add the remote to your local repository

GitHub will show you commands to use. Copy the one that looks like:

```
git remote add origin https://github.com/YOUR-USERNAME/atomic-movement-detection.git
```

Run this command in your terminal, replacing the URL with the one GitHub provides.

## 3. Push your code to GitHub

Push the code using:

```
git push -u origin master
```

Or if you're using `main` as your default branch:

```
git push -u origin main
```

The `-u` flag sets up tracking, so in the future you can simply run `git push`.

## 4. Verify on GitHub

Refresh your GitHub repository page to see your code on GitHub.

## Additional Git Commands

Here are some useful Git commands for maintaining your repository:

```
# See status of your changes
git status

# Create a new branch
git checkout -b new-feature

# Switch to an existing branch
git checkout branch-name

# Add specific files
git add file1.py file2.py

# Commit with a message
git commit -m "Description of changes"

# Push changes
git push

# Pull latest changes
git pull
``` 