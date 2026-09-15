from pydantic import BaseModel


class Settings(BaseModel):
    postgres_dsn: str = "postgresql+psycopg://streamforge:streamforge@localhost:5432/streamforge"
    kafka_bootstrap: str = "localhost:9092"
    kafka_topic: str = "bike-demand"
    source_path: str = "Dataset/SeoulBikeData.csv"


def get_settings() -> Settings:
    return Settings()
