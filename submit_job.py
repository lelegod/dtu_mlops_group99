from google.cloud import aiplatform

BUCKET_NAME = "gs://dtu-mlops-group99-data"

aiplatform.init(project="dtumlopsgroup99", location="europe-west1", staging_bucket=BUCKET_NAME)

job = aiplatform.CustomContainerTrainingJob(
    display_name="tennis-training-v1",
    container_uri="europe-west1-docker.pkg.dev/dtumlopsgroup99/dtu-mlops-images/train-image:latest",
    model_serving_container_image_uri="europe-docker.pkg.dev/vertex-ai/prediction/xgboost-cpu.1-6:latest",
)


job.run(
    model_display_name="tennis-model",
    machine_type="n1-standard-4",
    replica_count=1,
)
