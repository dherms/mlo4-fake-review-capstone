# mlo4-fake-review-capstone

To Run:

```bash
docker run --env-file .aws.env -p8000:8000 -p8001:8001 -p8002:8002 --rm --net=host nvcr.io/nvidia/tritonserver:22.09-py3 tritonserver --model-repository=s3://fourthbrain-fake-review-detector/models/
```