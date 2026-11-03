# Output Formats

The primary output of eris is a tabular TSV report detailing the evolutionary context of every assembled locus.

`{prefix}_report.gff` holds the report's genes as GFF3, each once, with the report's `gene_id` as its `ID`, so the
reported genes can be viewed or joined without the assembly-wide `{prefix}_assembly.gff`. It is skipped with `--no-gff`.

::: eris.io.ReportRow
    options:
      show_root_heading: false
      members: []