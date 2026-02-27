# Paths
REMOTE = gdrive:faculdade/transcriptions
LOCAL = /home/thiago/dev/transcriptions/
IGNORE_FILE = .rcloneignore

# -P: Show progress
# --transfers=4: Do 4 files at a time
# --exclude ".git/**": Always ignore the .git folder (prevents corruption/slowness)
# --exclude-from: Read patterns from your ignore file
FLAGS = -P --transfers=4 --exclude ".git/**" --exclude-from $(IGNORE_FILE)

.PHONY: help push force-push pull status

help:
	@echo "Available commands:"
	@echo "  make push        -> Upload changes (Safe - like git push)"
	@echo "  make force-push  -> Mirror local to remote (Destructive - like git push --force)"
	@echo "  make pull        -> Download changes (Safe - like git pull)"
	@echo "  make status      -> Show differences (like git status)"

push:
	rclone copy $(LOCAL) $(REMOTE) $(FLAGS)

force-push:
	@echo "⚠️WARNING: This will DELETE files on Google Drive that are not on your computer."
	@echo "Waiting 10 seconds... Press Ctrl+C to cancel."
	@sleep 10 
	rclone sync $(LOCAL) $(REMOTE) $(FLAGS)

pull:
	rclone copy $(REMOTE) $(LOCAL) $(FLAGS)

status:
	rclone check $(LOCAL) $(REMOTE) --one-way --exclude-from $(IGNORE_FILE) --exclude ".git/**"
