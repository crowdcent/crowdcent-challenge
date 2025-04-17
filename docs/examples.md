# Interactive Examples

## Fetching Challenges and Inference Data

This example demonstrates how to use the `ChallengeClient` to fetch available challenges and get details about the current inference data period. You can optionally provide your API key below, or leave it blank if you have the `CROWDCENT_API_KEY` environment variable or a `.env` file configured.

/// marimo-embed
    height: 750px
    mode: edit
    app_width: full

```python
@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Challenge Information""")
    return

# Cell 1: Input for API Key (optional)
@app.cell
def __():
    import marimo as mo
    api_key_input = mo.ui.text(
        placeholder="Leave blank to use env/dotenv",
        label="API Key (Optional)",
        kind="password",
    )
    # Display the input field
    api_key_input
    # Return it so other cells can use its value
    return api_key_input, mo

# Cell 2: Fetch and display challenges
@app.cell
def __(api_key_input, mo):
    from crowdcent_challenge import ChallengeClient, CrowdcentAPIError, AuthenticationError
    
    # Initialize placeholder text
    challenge_info = mo.md("Enter your API key above and click 'List Challenges' to begin.")
    
    # Create button for listing challenges
    list_challenges_button = mo.ui.button(label="List Challenges")
    
    # React to the button click (its value increments each time)
    if list_challenges_button.value > 0:
        # Get API key from input if provided, else None
        api_key = api_key_input.value or None
        try:
            # Initialize client
            client = ChallengeClient(api_key=api_key)
            challenges = client.list_challenges()
            
            if not challenges:
                challenge_info = mo.md("No active challenges found.")
            else:
                # Create a dropdown to select a challenge
                challenge_options = {c['name']: c['slug'] for c in challenges}
                challenge_dropdown = mo.ui.dropdown(
                    options=challenge_options,
                    label="Select a Challenge",
                    value=list(challenge_options.values())[0] if challenge_options else None
                )
                
                challenge_info = mo.vstack([
                    mo.md(f"**Found {len(challenges)} active challenges:**"),
                    challenge_dropdown
                ])
        except AuthenticationError:
            challenge_info = mo.md(
                f"""
                **Authentication Error!**

                Please ensure your `CROWDCENT_API_KEY` is set correctly
                in your environment or a `.env` file, or enter it above.
                """
            )
        except CrowdcentAPIError as e:
             challenge_info = mo.md(f"**API Error:** `{e}`")
        except Exception as e:
             challenge_info = mo.md(f"**Unexpected Error:** `{e}`")
    
    # Display the button and results
    mo.vstack([list_challenges_button, challenge_info])
    
    # Return values for the next cell
    return mo, AuthenticationError, ChallengeClient, CrowdcentAPIError, list_challenges_button

# Cell 3: Get information about selected challenge
@app.cell
def __(mo, api_key_input, list_challenges_button, AuthenticationError, ChallengeClient, CrowdcentAPIError):
    # Initialize placeholder for challenge details
    challenge_details = mo.md("Select a challenge above to view details.")
    
    # Only proceed if challenges have been loaded
    if list_challenges_button.value > 0 and "_5" in globals():
        challenge_dropdown = globals()["_5"].get("challenge_dropdown")
        
        if challenge_dropdown and challenge_dropdown.value:
            get_details_button = mo.ui.button(label="Get Challenge Details")
            
            if get_details_button.value > 0:
                # Get API key from input if provided, else None
                api_key = api_key_input.value or None
                try:
                    # Initialize client
                    client = ChallengeClient(api_key=api_key)
                    selected_slug = challenge_dropdown.value
                    
                    # Fetch challenge details
                    challenge = client.get_challenge(selected_slug)
                    
                    # Try to get current inference period
                    try:
                        current_period = client.get_current_inference_data(selected_slug)
                        period_info = f"""
                        **Current Inference Period:**
                        * Release Date: {current_period.get('release_date', 'N/A')}
                        * Submission Deadline: {current_period.get('submission_deadline', 'N/A')}
                        * Time Remaining: {current_period.get('time_remaining', 'N/A')}
                        """
                    except Exception as e:
                        period_info = f"""
                        **No Active Inference Period**
                        {str(e)}
                        """
                    
                    # Try to get latest training data
                    try:
                        latest_training = client.get_latest_training_dataset(selected_slug)
                        training_info = f"""
                        **Latest Training Dataset:**
                        * Version: {latest_training.get('version', 'N/A')}
                        * Upload Date: {latest_training.get('uploaded_at', 'N/A')}
                        """
                    except Exception as e:
                        training_info = f"""
                        **No Training Datasets Available**
                        {str(e)}
                        """
                    
                    challenge_details = mo.md(
                        f"""
                        ## {challenge.get('name', 'Unknown Challenge')}
                        
                        **Slug:** `{selected_slug}`
                        
                        **Description:** {challenge.get('description', 'No description available.')}
                        
                        **Status:** {'Active' if challenge.get('is_active', False) else 'Inactive'}
                        
                        **Dates:** {challenge.get('start_date', 'N/A')} to {challenge.get('end_date', 'N/A')}
                        
                        {period_info}
                        
                        {training_info}
                        """
                    )
                except AuthenticationError:
                    challenge_details = mo.md("**Authentication Error!** Please check your API key.")
                except CrowdcentAPIError as e:
                    challenge_details = mo.md(f"**API Error:** `{e}`")
                except Exception as e:
                    challenge_details = mo.md(f"**Unexpected Error:** `{e}`")
            
            result = mo.vstack([get_details_button, challenge_details])
        else:
            result = challenge_details
    else:
        result = challenge_details
    
    # Display the result
    result
    return result

```

///
