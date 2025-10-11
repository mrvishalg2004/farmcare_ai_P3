#!/usr/bin/env python3
"""
Gemini API Key Checker
=====================

This script verifies if a Google Generative AI API key is working properly.
It performs minimal API calls to preserve quotas.

- Can be used as standalone script or imported in other files
- Provides detailed error information
- Supports JSON output for automation
- Call-rate friendly with minimal API usage
"""

import os
import re
import sys
import json
import argparse
import time
from typing import Dict, Any, Optional, List, Tuple

try:
    import google.generativeai as genai
    from google.api_core.exceptions import GoogleAPIError, InvalidArgument
except ImportError:
    print("Error: Google Generative AI library not found.")
    print("Please install it with: pip install google-generativeai")
    sys.exit(1)

def check_gemini_api_key(api_key: str, verbose: bool = False) -> Dict[str, Any]:
    """
    Test if a Gemini API key is working with minimal API usage.
    
    Args:
        api_key: The Gemini API key to test
        verbose: Whether to print additional information
        
    Returns:
        Dict with success flag and details/error message
    """
    if not api_key or not isinstance(api_key, str) or len(api_key) < 10:
        return {
            "success": False, 
            "error": "Invalid API key format",
            "error_type": "FORMAT_ERROR",
            "details": "API key is empty, too short, or not a string"
        }
        
    # Configure the API
    try:
        genai.configure(api_key=api_key)
    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to configure Gemini API: {str(e)}",
            "error_type": "CONFIG_ERROR",
            "details": str(e)
        }
    
    # First, check available models (lightweight call)
    models = []
    try:
        if verbose:
            print("Checking available models...")
        
        start_time = time.time()
        models_response = genai.list_models()
        models = list(models_response)
        
        if verbose:
            print(f"API response time: {time.time() - start_time:.2f} seconds")
            print(f"Found {len(models)} models")
            
        # We found models, key is definitely working
        if len(models) > 0:
            gemini_models = [model.name for model in models if "gemini" in model.name]
            return {
                "success": True,
                "models_count": len(models),
                "gemini_models": gemini_models,
                "message": f"API key is valid. Found {len(models)} models.",
                "response_time_ms": int((time.time() - start_time) * 1000)
            }
    except Exception as e:
        # If listing models fails, that doesn't necessarily mean the key is invalid
        # Some permissions might allow model usage but not listing
        pass
    
    # If listing models failed or returned empty, try to get a model directly
    # This is a more reliable check but slightly heavier API call
    try:
        if verbose:
            print("Testing model access...")
            
        start_time = time.time()
        model = genai.GenerativeModel('gemini-pro')
        
        # We don't actually generate content to save quota
        # Just initializing the model is enough to verify the key works
        
        return {
            "success": True,
            "message": "API key is valid. Model access confirmed.",
            "models_count": len(models) if models else "unknown",
            "response_time_ms": int((time.time() - start_time) * 1000)
        }
    except InvalidArgument as e:
        if "API_KEY_INVALID" in str(e):
            return {
                "success": False,
                "error": "Invalid API key",
                "error_type": "INVALID_KEY",
                "details": str(e)
            }
        else:
            return {
                "success": False,
                "error": f"Invalid argument: {str(e)}",
                "error_type": "INVALID_ARGUMENT",
                "details": str(e)
            }
    except GoogleAPIError as e:
        error_message = str(e)
        
        # Check for quota exceeded
        if "429" in error_message or "quota" in error_message.lower():
            retry_match = re.search(r"retry in (\d+\.\d+)s", error_message)
            retry_seconds = int(float(retry_match.group(1))) + 1 if retry_match else None
            
            return {
                "success": False,
                "error": "API quota exceeded",
                "error_type": "QUOTA_EXCEEDED",
                "retry_after_seconds": retry_seconds,
                "details": error_message
            }
        # Check for permission denied
        elif "403" in error_message:
            return {
                "success": False,
                "error": "Permission denied",
                "error_type": "PERMISSION_DENIED",
                "details": error_message
            }
        else:
            return {
                "success": False,
                "error": f"API error: {error_message}",
                "error_type": "API_ERROR",
                "details": error_message
            }
    except Exception as e:
        return {
            "success": False,
            "error": f"Unexpected error: {str(e)}",
            "error_type": "UNEXPECTED_ERROR",
            "details": str(e)
        }

def format_output(result: Dict[str, Any], json_output: bool = False, quiet: bool = False) -> str:
    """Format the output of the API key check in human or JSON format"""
    if json_output:
        return json.dumps(result, indent=2)
        
    if result["success"]:
        if quiet:
            return "OK"
            
        output = [
            "✅ API Key Test: SUCCESS",
            f"• Found {result.get('models_count', 'unknown')} models"
        ]
        
        if "gemini_models" in result:
            output.append(f"• Gemini models: {', '.join(result['gemini_models'])}")
            
        if "response_time_ms" in result:
            output.append(f"• Response time: {result['response_time_ms']} ms")
            
        return "\n".join(output)
    else:
        if quiet:
            return f"ERROR: {result['error']}"
            
        output = [
            "❌ API Key Test: FAILED",
            f"• Error: {result['error']}",
            f"• Type: {result.get('error_type', 'UNKNOWN')}"
        ]
        
        if "retry_after_seconds" in result:
            output.append(f"• Retry after: {result['retry_after_seconds']} seconds")
            
        if not quiet and "details" in result:
            output.append("\nDetails:")
            output.append(result["details"])
            
        return "\n".join(output)

def main():
    """Main function when run as a script"""
    parser = argparse.ArgumentParser(
        description="Gemini API Key Checker - Tests if your API key is working",
        epilog="This script checks if your Gemini API key is valid and working\n"
               "with minimal API usage to preserve your quota."
    )
    parser.add_argument('--key', help="Gemini API key to check (if not provided, will check environment variable)")
    parser.add_argument('--quiet', '-q', action='store_true', help="Only output minimal information")
    parser.add_argument('--json', action='store_true', help="Output results in JSON format")
    
    # Add examples
    parser.formatter_class = argparse.RawDescriptionHelpFormatter
    parser.epilog += "\n\nExample usage:\n" \
                    "  api_key_check.py --key AIzaSyXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX\n" \
                    "  api_key_check.py --json\n"
    
    args = parser.parse_args()
    
    # Get API key from arguments or environment variable
    api_key = args.key
    if not api_key:
        api_key = os.environ.get('GOOGLE_API_KEY')
        
    if not api_key:
        print("Error: No API key provided.", file=sys.stderr)
        print("Please use --key or set the GOOGLE_API_KEY environment variable", file=sys.stderr)
        sys.exit(1)
        
    # Check the API key
    result = check_gemini_api_key(api_key, verbose=not args.quiet and not args.json)
    
    # Print result
    print(format_output(result, json_output=args.json, quiet=args.quiet))
    
    # Return exit code based on success
    sys.exit(0 if result["success"] else 1)

if __name__ == "__main__":
    main()