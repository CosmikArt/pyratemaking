# RatePlan

::: pyratemaking.RatePlan

::: pyratemaking.core.indication
    options:
      show_root_heading: true
      members:
        - Indication
        - ExpenseProvision
        - loss_ratio_indication
        - pure_premium_indication

::: pyratemaking.core.classification
    options:
      show_root_heading: true
      members:
        - ClassificationResult
        - classify

::: pyratemaking.core.implementation
    options:
      show_root_heading: true
      members:
        - ImplementationResult
        - implement_rate_change
        - apply_caps_floors
