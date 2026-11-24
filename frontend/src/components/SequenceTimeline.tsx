import { useI18n } from '../i18n/UiLanguage';
import { getConceptDisplayLabel } from '../presentation/conceptLabels';
import type { VisualResource } from '../types/api';

export function SequenceTimeline({ resources, index, played, completed, onSelect }: {
  resources: VisualResource[]; index: number; played: ReadonlySet<number>; completed: boolean;
  onSelect: (index: number) => void;
}) {
  const { t } = useI18n();
  return <ol className="sequence-timeline" aria-label={t('timeline')}>
    {resources.map((resource, position) => {
      const state = position === index && !completed ? 'timelineCurrent' : played.has(position) ? 'timelineCompleted' : 'timelinePending';
      const label = getConceptDisplayLabel(resource.concept_id, resource.language);
      return <li key={`${position}-${resource.concept_id}`}><button
        className={`timeline-item ${state}`} aria-current={state === 'timelineCurrent' ? 'step' : undefined}
        aria-label={t('timelineItem', { label, position: position + 1, total: resources.length, state: t(state) })}
        onClick={() => onSelect(position)}>
        <span aria-hidden="true">{state === 'timelineCompleted' ? '✓' : state === 'timelineCurrent' ? '●' : '○'}</span>
        <span lang={resource.language === 'LSM' ? 'es-MX' : 'en-US'}>{label}</span>
      </button></li>;
    })}
  </ol>;
}
