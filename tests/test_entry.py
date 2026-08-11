"""The single entry point that dispatches to the three subcommands.

Dispatch is the only thing tested here. Each subcommand's own behaviour is
covered by test_aux_cli.py and test_pipeline_e2e.py, and re-testing it through
the dispatcher would only duplicate those.
"""

import pytest

from cardiacprep import entry


def test_no_arguments_prints_usage_and_fails(capsys):
    # Exit code 2, not 0: running the bare command is a usage error, so a
    # script that forgets its arguments does not look like it succeeded.
    assert entry.main([]) == 2
    assert "Commands:" in capsys.readouterr().out


@pytest.mark.parametrize("flag", ["-h", "--help", "help"])
def test_help_succeeds_and_lists_every_command(flag, capsys):
    assert entry.main([flag]) == 0
    out = capsys.readouterr().out
    for name in entry.COMMANDS:
        assert name in out


def test_version_flag(capsys):
    from cardiacprep import __version__

    assert entry.main(["--version"]) == 0
    assert __version__ in capsys.readouterr().out


def test_unknown_command_is_a_usage_error(capsys):
    assert entry.main(["frobnicate"]) == 2
    err = capsys.readouterr().err
    assert "unknown command 'frobnicate'" in err
    assert "Commands:" in err


def test_dispatches_to_the_named_subcommand(monkeypatch):
    seen = {}

    def fake_main(argv):
        seen["argv"] = argv
        return 0

    module = pytest.importorskip("cardiacprep.subject_plots")
    monkeypatch.setattr(module, "main", fake_main)

    assert entry.main(["inspect", "--list", "--output", "somewhere"]) == 0
    assert seen["argv"] == ["--list", "--output", "somewhere"]


def test_subcommand_return_code_is_passed_through(monkeypatch):
    module = pytest.importorskip("cardiacprep.subject_plots")
    monkeypatch.setattr(module, "main", lambda argv: 3)

    assert entry.main(["inspect"]) == 3


@pytest.mark.parametrize("alias,target", sorted(entry.ALIASES.items()))
def test_aliases_resolve_to_a_real_command(alias, target):
    assert target in entry.COMMANDS


def test_american_spelling_is_accepted(monkeypatch):
    module = pytest.importorskip("cardiacprep.dataset_summary")
    monkeypatch.setattr(module, "main", lambda argv: 0)

    assert entry.main(["summarize"]) == 0


def test_every_command_module_is_importable_and_has_main():
    # Guards against a typo in COMMANDS: the modules are imported lazily, so
    # a wrong name would otherwise only surface when a user ran that command.
    import importlib

    for name, command in entry.COMMANDS.items():
        module = importlib.import_module(command.module)
        assert callable(module.main), f"{name} -> {command.module}.main"
