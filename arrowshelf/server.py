import subprocess
import sys
from pathlib import Path

def get_server_binary_path() -> Path:
    """
    Finds the path to the packaged Rust binary ('arrowshelfd'), handling
    both installed packages and local development environments. This function
    is platform-aware and looks for '.exe' on Windows.
    """
    if sys.platform == "win32":
        binary_name = "arrowshelfd.exe"
    else:
        binary_name = "arrowshelfd"

    install_path = Path(__file__).parent / "bin" / binary_name
    if install_path.exists():
        return install_path

    project_root = Path(__file__).parent.parent
    dev_path = project_root / "target" / "debug" / binary_name
    if dev_path.exists():
        return dev_path

    dev_path_release = project_root / "target" / "release" / binary_name
    if dev_path_release.exists():
        return dev_path_release
    
    raise FileNotFoundError(
        f"ArrowShelf server binary ('{binary_name}') could not be found. "
        "Please run 'maturin develop' to build it."
    )

def main():
    """
    Launches the compiled ArrowShelf daemon (arrowshelfd).
    """
    try:
        server_path = get_server_binary_path()
        print(f"--- Launching ArrowShelf Server from: {server_path} ---")
        subprocess.run([server_path], check=True)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
    except (subprocess.CalledProcessError, KeyboardInterrupt):
        print("\nArrowShelf server shut down.")
        sys.exit(0)

if __name__ == "__main__":
    main()