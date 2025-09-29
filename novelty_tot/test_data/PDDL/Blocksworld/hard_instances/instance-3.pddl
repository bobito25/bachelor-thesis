

(define (problem BW-rand-6)
(:domain blocksworld-4ops)
(:objects a b c d e f )
(:init
(handempty)
(on a b)
(on b e)
(on c a)
(on d c)
(ontable e)
(ontable f)
(clear d)
(clear f)
)
(:goal
(and
(on b c)
(on d a)
(on e d)
(on f e))
)
)


