"""Compatibility entry point for the active ResNet18 E2-E7 ladder.

The former generic-CNN implementation is preserved only in the versioned
archive. All active execution delegates to ``run_fedecai_ladder``.
"""

from fedecai.models.run_fedecai_ladder import main


if __name__ == "__main__":
    main()
