"""Initial database setup and default admin user seed script."""
import logging
from app.database.session import SyncSessionLocal, init_db
from app.models.models import User
from app.core.security import get_password_hash
from app.services.seed_data import seed_all

logger = logging.getLogger("ai_datascientist.initial_data")


def seed_default_user(db=None) -> None:
    """Auto-seed default demo admin user if missing or ensure valid credentials."""
    should_close = False
    if db is None:
        db = SyncSessionLocal()
        should_close = True

    try:
        from app.core.security import verify_password
        admin_user = db.query(User).filter(User.email == "admin@aidatascientist.io").first()
        if not admin_user:
            logger.info("No admin user found. Seeding default demo admin user...")
            admin_user = User(
                email="admin@aidatascientist.io",
                username="admin@aidatascientist.io",
                full_name="Admin User",
                hashed_password=get_password_hash("admin"),
                is_active=True,
            )
            db.add(admin_user)
            db.commit()
            logger.info("Default admin user created: admin@aidatascientist.io / admin")
        else:
            if not verify_password("admin", admin_user.hashed_password):
                logger.info("Updating admin user password to default 'admin'...")
                admin_user.hashed_password = get_password_hash("admin")
                admin_user.is_active = True
                db.commit()
                logger.info("Default admin user password synchronized to 'admin'.")
            else:
                logger.info("Default admin user exists with valid credentials.")
    except Exception as e:
        logger.error(f"Failed to seed default admin user: {e}")
        db.rollback()
    finally:
        if should_close:
            db.close()


def main() -> None:
    """Initialize DB tables and run full seed sequence."""
    init_db()
    seed_all()
    seed_default_user()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("Running initial data setup...")
    main()
    logger.info("Initial data setup complete.")
