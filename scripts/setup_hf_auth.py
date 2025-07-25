#!/usr/bin/env python3
"""
Hugging Face Authentication Setup Script for Glitcher
Helps users authenticate with Hugging Face to access gated models like Llama.
"""

import os
import sys
import subprocess
from typing import Optional, Dict, Any


def check_huggingface_hub_installed() -> bool:
    """Check if huggingface_hub is installed."""
    try:
        import huggingface_hub
        return True
    except ImportError:
        return False


def install_huggingface_hub():
    """Install huggingface_hub if not present."""
    print("📦 Installing huggingface_hub...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub"])
        print("✅ huggingface_hub installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install huggingface_hub: {e}")
        return False


def get_current_token() -> Optional[str]:
    """Get current HF token from environment or HF cache."""
    # Check environment variable first
    token_env = os.getenv('HF_TOKEN') or os.getenv('HUGGING_FACE_HUB_TOKEN')
    if token_env:
        return token_env

    # Try to get from huggingface_hub cache
    try:
        from huggingface_hub import HfFolder
        return HfFolder.get_token()
    except:
        return None


def validate_token(token: str) -> Dict[str, Any]:
    """Validate HF token and get user info."""
    try:
        from huggingface_hub import whoami

        user_info = whoami(token=token)
        return {
            "valid": True,
            "username": user_info.get("name", "Unknown"),
            "email": user_info.get("email", "Unknown"),
            "orgs": user_info.get("orgs", []),
        }
    except Exception as e:
        return {
            "valid": False,
            "error": str(e)
        }


def test_gated_model_access(token: str, model_id: str = "meta-llama/Llama-3.2-1B-Instruct") -> Dict[str, Any]:
    """Test access to a gated model."""
    try:
        from huggingface_hub import model_info

        info = model_info(model_id, token=token)
        return {
            "accessible": True,
            "model_id": model_id,
            "model_name": info.modelId,
            "gated": getattr(info, 'gated', False),
        }
    except Exception as e:
        error_msg = str(e)
        if "401" in error_msg or "gated" in error_msg.lower():
            return {
                "accessible": False,
                "model_id": model_id,
                "error": "Access denied - you may need to request access to this model",
                "gated": True,
            }
        else:
            return {
                "accessible": False,
                "model_id": model_id,
                "error": error_msg,
                "gated": False,
            }


def login_interactive():
    """Interactive HF login process."""
    try:
        from huggingface_hub import login

        print("\n🔐 Starting Hugging Face authentication...")
        print("You'll need a Hugging Face token with access to gated models.")
        print("Get one at: https://huggingface.co/settings/tokens")
        print()

        # Try interactive login
        login()
        return True

    except Exception as e:
        print(f"❌ Login failed: {e}")
        return False


def login_with_token(token: str):
    """Login with provided token."""
    try:
        from huggingface_hub import login

        login(token=token)
        return True

    except Exception as e:
        print(f"❌ Login with token failed: {e}")
        return False


def save_token_to_env_file(token: str, env_file: str = ".env"):
    """Save token to .env file."""
    try:
        # Read existing .env file
        env_lines = []
        if os.path.exists(env_file):
            with open(env_file, 'r') as f:
                env_lines = f.readlines()

        # Remove existing HF_TOKEN lines
        env_lines = [line for line in env_lines if not line.startswith('HF_TOKEN=')]

        # Add new HF_TOKEN
        env_lines.append(f'HF_TOKEN={token}\n')

        # Write back to file
        with open(env_file, 'w') as f:
            f.writelines(env_lines)

        print(f"✅ Token saved to {env_file}")
        return True

    except Exception as e:
        print(f"❌ Failed to save token to {env_file}: {e}")
        return False


def print_status_report():
    """Print comprehensive authentication status."""
    print("=" * 80)
    print("HUGGING FACE AUTHENTICATION STATUS")
    print("=" * 80)

    # Check if huggingface_hub is installed
    if not check_huggingface_hub_installed():
        print("❌ huggingface_hub not installed")
        if install_huggingface_hub():
            print("✅ huggingface_hub installed successfully")
        else:
            print("❌ Failed to install huggingface_hub")
            return
    else:
        print("✅ huggingface_hub is installed")

    # Check current token
    current_token = get_current_token()
    if not current_token:
        print("❌ No Hugging Face token found")
        print("   Set HF_TOKEN environment variable or run 'huggingface-cli login'")
        return

    print(f"✅ Token found: {current_token[:8]}..." if len(current_token) > 8 else "✅ Token found")

    # Validate token
    print("\n🔍 Validating token...")
    validation = validate_token(current_token)

    if validation["valid"]:
        print(f"✅ Token is valid")
        print(f"   Username: {validation['username']}")
        print(f"   Email: {validation['email']}")
        if validation["orgs"]:
            print(f"   Organizations: {', '.join([org['name'] for org in validation['orgs']])}")
    else:
        print(f"❌ Token validation failed: {validation['error']}")
        return

    # Test gated model access
    print("\n🚪 Testing gated model access...")
    test_models = [
        "meta-llama/Llama-3.2-1B-Instruct",
        "meta-llama/Llama-3.2-3B-Instruct",
        "meta-llama/Meta-Llama-3-8B-Instruct",
    ]

    accessible_models = []
    inaccessible_models = []

    for model_id in test_models:
        print(f"   Testing {model_id}...")
        result = test_gated_model_access(current_token, model_id)

        if result["accessible"]:
            print(f"   ✅ {model_id} - Accessible")
            accessible_models.append(model_id)
        else:
            print(f"   ❌ {model_id} - {result['error']}")
            inaccessible_models.append(model_id)

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    if accessible_models:
        print(f"✅ You have access to {len(accessible_models)} models:")
        for model in accessible_models:
            print(f"   • {model}")

    if inaccessible_models:
        print(f"\n⚠️  You need to request access to {len(inaccessible_models)} models:")
        for model in inaccessible_models:
            print(f"   • {model}")
            print(f"     Request access at: https://huggingface.co/{model}")

    if not accessible_models:
        print("❌ No accessible gated models found")
        print("   You may need to:")
        print("   1. Request access to models at https://huggingface.co/")
        print("   2. Generate a new token with appropriate permissions")
        print("   3. Wait for access approval (can take time)")
    else:
        print("\n🎉 Authentication successful! You can use Glitcher with gated models.")

    print("\n" + "=" * 80)


def main():
    """Main function."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Hugging Face Authentication Setup for Glitcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python setup_hf_auth.py --status                    # Check current auth status
  python setup_hf_auth.py --login                     # Interactive login
  python setup_hf_auth.py --token YOUR_TOKEN          # Login with token
  python setup_hf_auth.py --token YOUR_TOKEN --save   # Login and save to .env
        """
    )

    parser.add_argument("--status", action="store_true",
                       help="Check authentication status")
    parser.add_argument("--login", action="store_true",
                       help="Interactive login process")
    parser.add_argument("--token", type=str,
                       help="Login with specific token")
    parser.add_argument("--save", action="store_true",
                       help="Save token to .env file")
    parser.add_argument("--env-file", type=str, default=".env",
                       help="Path to .env file (default: .env)")
    parser.add_argument("--test-model", type=str,
                       default="meta-llama/Llama-3.2-1B-Instruct",
                       help="Model to test access (default: meta-llama/Llama-3.2-1B-Instruct)")

    args = parser.parse_args()

    # Install huggingface_hub if needed
    if not check_huggingface_hub_installed():
        print("📦 huggingface_hub not found, installing...")
        if not install_huggingface_hub():
            print("❌ Failed to install huggingface_hub. Please install manually:")
            print("   pip install huggingface_hub")
            sys.exit(1)

    # Handle different commands
    if args.status or (not args.login and not args.token):
        # Default action is to show status
        print_status_report()

    elif args.login:
        # Interactive login
        if login_interactive():
            print("✅ Login successful!")
            if args.save:
                token = get_current_token()
                if token:
                    save_token_to_env_file(token, args.env_file)
        else:
            print("❌ Login failed")
            sys.exit(1)

    elif args.token:
        # Login with specific token
        print(f"🔐 Logging in with provided token...")

        # Validate token first
        validation = validate_token(args.token)
        if not validation["valid"]:
            print(f"❌ Invalid token: {validation['error']}")
            sys.exit(1)

        print(f"✅ Token is valid for user: {validation['username']}")

        # Login with token
        if login_with_token(args.token):
            print("✅ Login successful!")

            # Save to .env if requested
            if args.save:
                save_token_to_env_file(args.token, args.env_file)

            # Test model access
            print(f"\n🧪 Testing access to {args.test_model}...")
            result = test_gated_model_access(args.token, args.test_model)

            if result["accessible"]:
                print(f"✅ Access confirmed to {args.test_model}")
            else:
                print(f"⚠️  No access to {args.test_model}: {result['error']}")
                print(f"   Request access at: https://huggingface.co/{args.test_model}")
        else:
            print("❌ Login failed")
            sys.exit(1)


if __name__ == "__main__":
    main()
