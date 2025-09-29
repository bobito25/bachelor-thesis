import os
import subprocess


def validate_plan(domain, instance, pddl_plan, verbose=False) -> bool:
    val_path = os.getenv("VAL")
    validator_executable = val_path  # os.path.join(val_path, "bin", "validate", "validate.exe")

    # write the plan to a file with random name
    random_str = os.urandom(8).hex()
    plan_file = f"plan_{random_str}.pddl"
    with open(plan_file, "w") as f:
        f.write(pddl_plan)

    cmd = [validator_executable, domain, instance, plan_file]
    if verbose:
        print(f"Running validation command: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)

        if verbose:
            print("Validator stdout:")
            print(result.stdout)
            if result.stderr:
                print("Validator stderr:")
                print(result.stderr)

        response = result.stdout + result.stderr
        if 'Problem in domain' in response:
            raise Exception('Problem in domain: Check PDDL Writer')
        return "Plan valid" in response
    except FileNotFoundError:
        print(f"Error: The validator executable was not found at '{validator_executable}'")
        return False
    except subprocess.TimeoutExpired:
        print("Error: The validator took too long to respond.")
        return False
    finally:
        # Clean up the plan file
        if os.path.exists(plan_file):
            os.remove(plan_file)
