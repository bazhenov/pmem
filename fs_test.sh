#!/usr/bin/env bash
set -e

mount -t nfs -o nolocks,vers=3,tcp,port=11111,mountport=11111,soft 127.0.0.1:/ mnt/
trap 'umount mnt' EXIT

run_test() {
  local test_name="$1"
  local mutation_function="$2"
  local predicate="$3"
  local error_message="$4"

  printf '%30s...' "$test_name"
  sh -c "$mutation_function"
  if ! eval $predicate; then
  echo -e "\e[31mFailed\e[0m" >&2
    echo "$error_message" >&2
    exit 1
  fi
  echo -e "\e[32mOk\e[0m"
}

run_test "Creating directory" "mkdir mnt/dir" "[ -d 'mnt/dir' ]" "Directory 'mnt/dir' does not exist"
run_test "Removing directory" "rmdir mnt/dir" "! [ -d 'mnt/dir' ]" "Directory 'mnt/dir' still exist"

run_test "Create file" "touch mnt/file" "[ -f 'mnt/file' ]" "File 'mnt/file' does not exist"
run_test "Check file content" "echo 'Hello world' > mnt/file" "grep -q 'Hello world' mnt/file"\
    "File content is incorrect"
run_test "Copy file" "cp mnt/file mnt/file2" "[ -f 'mnt/file2' ]" "File 'mnt/file2' does not exist"
run_test "Check file content" "echo 'Hello world' > mnt/file2" "grep -q 'Hello world' mnt/file2"\
    "File content is incorrect"
run_test "Delete file" "rm -f mnt/file" "! [ -f 'mnt/file' ]" "File 'mnt/file' still exist"
run_test "Delete file" "rm -f mnt/file2" "! [ -f 'mnt/file2' ]" "File 'mnt/file2' still exist"
