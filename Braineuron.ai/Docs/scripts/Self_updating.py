import time

CHECK_INTERVAL = 300  # check every 5 minutes

def update_and_retrain_loop(model, numeric_data, text_data, targets):
    while True:
        updated = False
        for url in REPO_URLS:
            repo_name = url.rstrip("/").split("/")[-1]
            repo_path = os.path.join("repos", repo_name)
            pull_status = os.system(f"cd {repo_path} && git pull")
            if pull_status != 0:
                print(f"⚠️ Repo {repo_name} pull failed or updated")
            else:
                updated = True  # assume change
        if updated:
            print("🔄 Changes detected. Rebuilding numeric/text data and retraining NeoMind...")
            # Rebuild data
            numeric_data_new = build_numeric_data()
            text_data_new, _ = build_text_data()
            train(model, numeric_data_new, text_data_new, targets, epochs=10, batch_size=32)
            print("✅ NeoMind retraining complete.")
        else:
            print("⏳ No changes detected.")
        time.sleep(CHECK_INTERVAL)