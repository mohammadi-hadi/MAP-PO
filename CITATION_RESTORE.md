Restore after review

The citation metadata is withheld while the paper is under double-blind review,
because anonymous.4open.science mirrors CITATION.cff verbatim: it exempts the
file from term redaction, so the author name and ORCID stay readable in the
anonymized copy.

On acceptance, restore CITATION.cff from git history:

    git show 7d461d7:CITATION.cff > CITATION.cff

then update the title and repository-code fields and re-enable
.github/workflows/citation.yml.
