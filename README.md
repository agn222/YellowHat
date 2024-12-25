# Complete Git and Pull Request Tutorial (Git Together)

Welcome to the comprehensive Git and Pull Request tutorial for the Xplore project! This guide will walk you through the entire process of contributing to the project, from forking the repository to creating a pull request. We'll cover basic and advanced Git commands, error handling, and best practices.
name : Dhawallmate 
## Table of Contents

1. [Getting Started](#getting-started)
2. [Basic Git Commands](#basic-git-commands)
3. [Making Changes](#making-changes)
4. [Advanced Git Commands](#advanced-git-commands)
5. [Creating a Pull Request](#creating-a-pull-request)
6. [Troubleshooting](#troubleshooting)
7. [Best Practices](#best-practices)
8. [Interactive Practice](#interactive-practice)

## Getting Started

### 1. Fork the Repository

1. Visit the [Xplore repository](https://github.com/SharanRP/Xplore) on GitHub.
2. Click the "Fork" button in the top-right corner to create a copy in your account.

![Forking a Repository](https://github.com/SharanRP/Xplore/assets/136159249/caaa1619-d5c7-4108-b60b-16de8d26884f)

### 2. Clone Your Fork

Clone your forked repository to your local machine:

```bash
git clone https://github.com/YOUR_USERNAME/Xplore.git
```

Replace `YOUR_USERNAME` with your GitHub username.

### 3. Navigate to the Project Directory

```bash
cd Xplore
```

### 4. Set Up Remote

Add the original repository as a remote to keep your fork updated:

```bash
git remote add upstream https://github.com/SharanRP/Xplore.git
```

## Basic Git Commands

### Checking Status

View the status of your working directory:

```bash
git status
```

### Creating a New Branch

Create and switch to a new branch:

```bash
git checkout -b feature/your-feature-name
```

### Viewing Branches

List all branches:

```bash
git branch
```

### Switching Branches

Switch to an existing branch:

```bash
git checkout branch-name
```

## Making Changes

1. Make your desired changes to the codebase.
2. Stage your changes:
   ```bash
   git add .
   ```
3. Commit your changes:
   ```bash
   git commit -m "Add feature xyz"
   ```
4. Push your changes to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

## Advanced Git Commands

### Stashing Changes

Temporarily store changes:

```bash
git stash
```

Apply stashed changes:

```bash
git stash pop
```

### Fetching Updates

Fetch updates from the original repository:

```bash
git fetch upstream
```

### Merging Changes

Merge updates from the original repository:

```bash
git checkout main
git merge upstream/main
```

### Rebasing

Rebase your feature branch:

```bash
git checkout feature/your-feature-name
git rebase main
```

### Using git reset

`git reset` is a powerful command for undoing changes and managing your commit history. Use it carefully to avoid losing work.

#### Types of reset:

1. **Soft reset** (`--soft`): Moves HEAD but doesn't change staging area or working directory.
2. **Mixed reset** (`--mixed` or default): Moves HEAD and updates staging area.
3. **Hard reset** (`--hard`): Moves HEAD, updates staging area and working directory.

#### Common use cases:

- Undo last commit (keep changes): `git reset HEAD~1`
- Undo multiple commits: `git reset HEAD~n` (replace `n` with number of commits)
- Reset to specific commit: `git reset commit-hash`
- Unstage changes: `git reset HEAD file-name`
- Discard all local changes (use cautiously): `git reset --hard HEAD`

## Creating a Pull Request

1. Go to your forked repository on GitHub.
2. Switch to your new branch.
3. Click "New Pull Request".
4. Select the base repository and branch you want to merge into.
5. Provide a title and description for your changes.
6. Click "Create Pull Request".

![Create Pull Request](https://github.com/SharanRP/Xplore/assets/136159249/d0c7383e-9dae-436d-af27-f5728ec90e6d)

## Troubleshooting

### Common Errors and Solutions

1. **"fatal: not a git repository"**
   - Solution: Ensure you're in the correct directory or initialize with `git init`.

2. **"fatal: remote origin already exists"**
   - Solution: Update the remote URL:
     ```bash
     git remote set-url origin https://github.com/YOUR_USERNAME/Xplore.git
     ```

3. **"error: failed to push some refs"**
   - Solution: Pull the latest changes before pushing:
     ```bash
     git pull origin main
     ```

4. **"You are in 'detached HEAD' state"**
   - Solution: Return to the original branch:
     ```bash
     git checkout branch-name
     ```

## Best Practices

1. Keep commits small and focused on a single change.
2. Write clear, concise commit messages.
3. Pull the latest changes from upstream before starting new work.
4. Use branches for new features or bug fixes.
5. Review your changes before committing.
6. Test your changes thoroughly before creating a pull request.

## Interactive Practice

Try this exercise to practice the Git workflow:

1. Create a new branch called `practice/your-name`.
2. Create a new file called `your-name.md` with some content about yourself.
3. Stage and commit this file.
4. Push the branch to your fork.
5. Create a pull request for this branch.
6. Make a change to your file, commit, and push again.
7. Use `git reset` to undo the last commit while keeping the changes.
8. Recommit the changes with a different commit message.
9. Force push the branch to update your pull request.

## Adding Your Information

Please add your name and registration number below:

- Name: [Your Name]
- Registration Number: [Your Registration Number]

Congratulations on completing this comprehensive Git and Pull Request tutorial! Remember, practice is key to mastering Git. Don't hesitate to experiment in a test repository to gain more confidence with these commands.

If you have any questions or encounter any issues, please don't hesitate to ask for help. Happy coding!
"# INH-git" 
