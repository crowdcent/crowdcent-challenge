## Scheduling a Kaggle notebook
If you're just starting out, we recommend using Kaggle Notebooks to schedule your submissions.

1. **Settings (⚙) → Schedule a notebook run → On**  
2. Choose **Frequency** (daily / weekly / monthly), **Start date**, **Runs ≤ 10** → **Save**  
3. A clock icon appears; each run writes a new **Version** with full logs & outputs  
4. Limits: **CPU-only • ≤ 9 h per run • 1 private / 5 public schedules active**  
5. Pause or delete the job anytime from the same Settings card  

<sub>Need GPUs? Trigger notebook commits with the Kaggle API from cron/GitHub Actions.</sub>

## Scheduling a Google Colab (Vertex AI) notebook
https://www.youtube.com/watch?v=ypGah2gRYck

1. **Create a Google Cloud account** if you don't have one already
2. **Go to [Google Colab Notebooks in Vertex AI](https://console.cloud.google.com/vertex-ai/colab/notebooks)**
3. **Set up a schedule:**
   - Open your notebook in Colab
   - Click **Runtime → Manage sessions**
   - Select **Recurring** and configure your schedule
   - Set frequency (daily/weekly/monthly) and duration
   - Click **Save**
4. **Authentication options:**
   - Use service account keys stored securely
   - Set up environment variables in the Vertex AI console
   - Use Google Cloud's Secret Manager for API keys

<sub>Note: Scheduled Colab notebooks run on Google Cloud and may incur charges based on your usage.</sub>