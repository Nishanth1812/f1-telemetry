$domains = @('saedigitallibrary.com','www.sae.org','pirelli.com','formula1.com','fia.com','tiretechnology.com','formulaone.com','motorsporttech.com','tire-rack.com','continental.com','apracing.com','brembo.com','hitco.com','goodyear.com','michelin.com','cooper-tires.com','bbs.de','racingpoint.co.uk','mercedesamgf1.com','redbullracing.com','ferrari.com','mclarenf1.com','astonmartinf1.com','redbull.com','alpine-f1team.com','williamsf1.com','alphataurif1.com','haasf1team.com','stakef1team.com','alphatauri.com')
foreach ($d in $domains) {
    try {
        $resp = Invoke-WebRequest -Uri "https://www.$d/" -UseBasicParsing -TimeoutSec 12 -ErrorAction Stop
        Write-Host "${d}: 200"
    } catch {
        Write-Host "${d}: FAIL"
    }
}
