from google.cloud import automl_v1beta1 as automl

automl_client = automl.AutoMlClient()

# Get the full path of the model.
model_full_id = automl_client.model_path(
    'balmy-coral-222122', 'us-central1', 'ICN7342185618183620124'
)

# Create client for prediction service.
prediction_client = automl.PredictionServiceClient()

# Read the image and assign to payload.
with open("C:\Dev\GarbageClassification\dataset-original\dataset-original\cardboard\cardboard1.jpg", "rb") as image_file:
    content = image_file.read()
payload = {"image": {"image_bytes": content}}

# params is additional domain-specific parameters.
# score_threshold is used to filter the result
# Initialize params
score_threshold = '0.7'
params = {}
if score_threshold:
    params = {"score_threshold": score_threshold}

response = prediction_client.predict(model_full_id, payload, params)
print("Prediction results:")
for result in response.payload:
    print("Predicted class name: {}".format(result.display_name))
    print("Predicted class score: {}".format(result.classification.score))