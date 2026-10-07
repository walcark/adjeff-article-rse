from adjeff_article_1 import get_auxdata_path


def main():
    str_auxdata_path: str = str(get_auxdata_path())
    print(f"export SMARTG_DIR_AUXDATA={str_auxdata_path}")


if __name__ == "__main__":
    main()
